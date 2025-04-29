import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.StringTokenizer;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.filechooser.FileNameExtensionFilter;

public class PanelEmpresa extends JPanel {

	private JFileChooser archivo,
						 guardar;
	private JButton btimportar,
					btregresar,
					btExportar;
	private Persona persona;
	private Deduccion deducciones;
	private File f,
				 gg;
	
	public PanelEmpresa(){
		
		super();
		this.setPreferredSize(new Dimension(450,200));
		this.setBackground(Color.lightGray);
		JLabel mensaje=new JLabel("             Por favor importe aquí su archivo         ");
		mensaje.setFont(new Font("TimesRoman", Font.BOLD,15));
		this.add(mensaje);
		this.add(new JLabel("                                               "
				+ "                                                                   "
				+ "                                                                   "));
		
		this.btimportar=new JButton("Importar .csv o .xls");
		this.add(btimportar);
		this.btimportar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				FileNameExtensionFilter filt=new FileNameExtensionFilter("CSV File","csv");	
				//FileNameExtensionFilter filt2=new FileNameExtensionFilter("XLS File","xls");
				archivo=new JFileChooser();
				archivo.setFileFilter(filt);
				//archivo.setFileFilter(filt2);
				archivo.showOpenDialog(null);	
				f=archivo.getSelectedFile();
				
			}
		});
		this.btExportar=new JButton("Exportar resultados");
		this.add(this.btExportar);
		this.add(new JLabel("                                     "
				+ "                                                         "));
		this.btExportar.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				guardar=new JFileChooser();
				guardar.showSaveDialog(null);
				Leer();
				
			}
		});
		this.btregresar=new JButton("Regresar a menú principal");
		this.add(this.btregresar);
		this.btregresar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				VentanaInicial v1=new VentanaInicial();
			}
		});
		
		
	}public void Leer()  {
		FileReader fr;
		File gg=this.guardar.getSelectedFile();
		
		try {
			fr = new FileReader(f);
			PrintWriter pw=new PrintWriter(gg+".csv");
			pw.println("Nombre"+","+"RFC"+","+"Sueldo Mensual"+","+"Ingreso Anual"+","+"Aguinaldo"+","
			+ "Aguinaldo Excento"+","+"Aguinaldo Gravado"+","+"Prima Vacacional"+","+"Vacacional Excento"
			+","+"Vacacional Gravada"+","+"Ingresos Gravados"+","+"Medicos"+","+"Funerarios"+","+"SGMM"
			+","+"Hipoteca"+","+"Donativos"+","+"Subcuenta"+","+"Transporte Escolar"+","+"Nivel Educativo"+
			","+"Max Deduccion Colegiatura"+","+"Colegiatura Pagada"+","+"Total Deduccion"+","+
			"Deducciones Permitidas"+","+"Monto ISR"+","+"Cuota Fija"+","+"Porcentaje Excedido"+","+"Pago Excedido"+","+"Total Pagar");
			BufferedReader br=new BufferedReader(fr);
			StringTokenizer sr;
			String linea;
			try {
			while((linea=br.readLine())!=null) {
				sr=new StringTokenizer(linea);
				String Nombre=sr.nextToken();
				sr.nextToken();
				String RFC=sr.nextToken();
				sr.nextToken();
				double sueldoMensual=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double aguinaldo=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double primaVaca=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double medHosp=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double funerarios=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double SGM=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double hipoteca=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double donativos=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double subRetiro=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				double Transporte=Double.parseDouble(sr.nextToken());
				sr.nextToken();
				String educacion=sr.nextToken();
				sr.nextToken();
				double Colegiatura=Double.parseDouble(sr.nextToken());
				persona=new Persona(Nombre,RFC,sueldoMensual,aguinaldo,primaVaca,
							educacion,Colegiatura,SGM,funerarios,medHosp,hipoteca,
							donativos,subRetiro,Transporte);
				deducciones=new Deduccion(persona);
				pw.println(persona.Nombre+","+persona.RFC+","+persona.sueldoMensual+","+persona.ingresoAnual()+
						","+persona.aguinaldo+","+persona.aguinaldoExento()+","+persona.aguinaldoGravado()+","+
						persona.primaVacacional+","+persona.primaExenta+","+persona.primaGravado()+","+
						persona.getTIngresosGravan()+","+persona.medHosp+","+persona.funerarios+","+persona.SGM+","+
						persona.hipoteca+","+persona.donativos+","+persona.subRetiro+","+persona.Transporte+","+
						persona.educacion+","+persona.dEducacion+","+persona.colegiatura+","+deducciones.deduccion()+","+
						deducciones.dPermitida()+","+deducciones.montoISR()+","+deducciones.cuotaFija+","+deducciones.porcentaje+","+
						deducciones.pagoEx+","+deducciones.totalPagar());

			}
			br.close();
			pw.close();
} catch (IOException e1) {
			e1.printStackTrace();
}
			
			
			
			
			
		
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}
	

	
}
