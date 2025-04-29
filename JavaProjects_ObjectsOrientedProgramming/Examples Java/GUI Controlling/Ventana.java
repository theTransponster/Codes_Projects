import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JFrame;

//paquete grafico swing(paquete qe trata de ejecutarse de la misma manera en cualquier plataforma) 
//y awt(llamadas nativas al sistema operativo)
//frames tienen un ciclo de vida (ciertos metodos que se mandan a llamar en circunstancias especificas en la 
//vida del objeto)por ejemplo el metodo paint y se manda a llaamr automaticamente
//Cada que considera que se tienen que repintar, por ejemplo cuando le hago un cambio de tamaño a la venta

public class Ventana extends JFrame {
	public Ventana(){
		super("Mi primer ventana en Java");
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setLocation(300,50);
		PanelDibujo mipanel=new PanelDibujo();
		this.add(mipanel); //lo agregas al contenedor de la ventana
		this.add(new ControlesPocoyo(mipanel),BorderLayout.WEST);
		this.pack(); //se ajusta al tamaño de sus componentes
		this.setVisible(true);
		
	}
	
	public static void main(String[] args) {
		
		//ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		//ventana.setVisible(true); //se hace la ventana visible /si cierro la ventana no por eso deja de correr el programa/ que la ultima instruccion sea el set visible porque primero preparas la ventan y luego la muestras 
		Ventana miVentanita=new Ventana();
	}

}
