import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;


public class PanelInicial extends JPanel{

	private JButton btEmpresa,
	    			btIndividual;
	private JFrame topFrame;
	public PanelInicial(){
		super();
		//topFrame=(JFrame)SwingUtilities.getWindowAncestor(this);
		
		//topFrame=(JFrame)(this.getBorderLayout());
		
		this.setPreferredSize(new Dimension(550,300));
		this.setBackground(Color.white);
		 
		JLabel bienvenido=new JLabel("                         Bienvenido                   ");
		bienvenido.setForeground(Color.BLUE);
		bienvenido.setFont(new Font("TimesRoman", Font.BOLD,18));
		this.add(bienvenido);
		this.add(new JLabel("                                                                         "
				+ "                                                                                   "
				+ "                                                                                   "));
		
		this.setFont(new Font("TimesRoman", Font.BOLD,15));
		this.add(new JLabel("                                    Para calcular su ISR seleccione modalidad:                           "
				+ "                                                     "));
		this.add(new JLabel("                                                                         "
				+ "                                                                                   "
				+ "                                                                                   "));
		this.add(new JLabel("                            Persona Física               "));
		this.add(new JLabel("           Base de Datos Empleados             "));
		this.btIndividual=new JButton("Individual");
		this.add(btIndividual);
		this.btIndividual.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent arg0) {
				VentanaIndividual v=new VentanaIndividual();
			}	
		});
		this.add(new JLabel("                     "));
		this.btEmpresa=new JButton("Empresa");
		this.add(this.btEmpresa);
		this.btEmpresa.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				
				VentanaEmpresa v2=new VentanaEmpresa();
				//topFrame.setDefaultCloseOperation(ABORT);
			}
		});
	}
	
	
}
